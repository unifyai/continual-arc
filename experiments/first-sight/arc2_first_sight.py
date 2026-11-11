"""First-sight solve rate: how often a model solves a track's fresh instances from three demonstrations.

An item is one instance of a task: three demonstration pairs and a test input, drawn by the source as an
instance and its first demonstration request, so a model meets the task as the demo-cache floor first does.
The model sees them in a fixed prompt (:func:`prompt`) and must reply with the output grid, which must match
exactly. One answer, no feedback, no second attempt.

The items are a calibration sample that a track of the benchmark writes (the repository's
``scripts/arc2_calibration_sample.py``: each item one fresh instance of a stream task with three demonstrations, its
``k`` below ``--instances`` kept). Calls go to OpenRouter from a Modal container:

    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... OPENROUTER_API_KEY=... \
        uv run --with modal python arc2_first_sight.py --items SAMPLE --model MODEL --effort medium --instances 4

sends the items to one Modal function (in the Modal environment ``MODAL_ENVIRONMENT``, Modal's default unless set),
and writes every answer, its charge, its provider and reasoning tokens (:func:`ask`) and each task's solve rate to
``reports/<sample>-<model>-<effort>.json`` beside this file.

``--skip K`` asks only each task's instances from the K-th on and keeps the answers on file to the ones before, so
that a reading is extended with fresh instances, never asked twice.

No two items open alike past the instruction sentence, which is shorter than any provider caches, since every item
draws its own instance and demonstrations, so there is no prompt to cache.

``--providers SLUG,...`` sends every call to those OpenRouter providers alone, in that order with no fallback (a slug
may name one endpoint of a provider, as ``moonshotai/mxfp4`` does), and writes ``-on-<slugs>`` into the report's name
beside the routed reading. OpenRouter otherwise spreads a model's calls over every provider serving it, which for an
open model can be a dozen, each with its own quantization and its own mapping of a reasoning effort, so a reading taken
on one day's routing need not hold on another's.

This is the ``--items`` mode of the calibration that measured the private set's rules at first sight (the paper's
Appendix J); its modes that read the private rules are not here. Table tab:first-sight's ARC-AGI-2 columns were
measured with it on 28 October 2026, when it waited 900 seconds rather than 1,800 for a reply, recorded no provider,
served model, generation id or reasoning tokens, and had no ``--providers``, ``--skip`` or ``--workers``: its
requests were these without ``--providers``. Appendix K's reading of Kimi K3.1 Pro's effort is this file with
``--providers``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

N_DEMOS = 3
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
REPORTS = Path(__file__).resolve().parent / "reports"


def _rows(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)


def prompt(item: dict) -> str:
    parts = [("Each grid below is a list of rows, and each digit 0 to 9 is a colour. Every example shows an input "
              "grid and the output grid that one fixed rule makes from it. Work out the rule, then apply it to the "
              "test input.")]
    for n, pair in enumerate(item["demos"], 1):
        parts.append(f"Example {n}\ninput:\n{_rows(pair['input'])}\noutput:\n{_rows(pair['output'])}")
    parts.append(f"Test\ninput:\n{_rows(item['input'])}")
    parts.append("Reply with the test output grid only: one row per line, the digits separated by spaces, "
                 "nothing else.")
    return "\n\n".join(parts)


_ROW = re.compile(r"^\s*(?:\d\s+)*\d\s*$")


def parse(text: str) -> list[list[int]] | None:
    """The last block of digit rows in ``text``, as a rectangular grid, or None."""
    blocks: list[list[list[int]]] = []
    current: list[list[int]] = []
    for line in (text or "").splitlines():
        line = line.strip().strip("`").strip()
        if _ROW.match(line):
            row = [int(t) for t in line.split()] if " " in line else [int(ch) for ch in line]
            current.append(row)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    for block in reversed(blocks):
        if len({len(row) for row in block}) == 1:
            return block
    return None


def ask(item: dict, model: str, effort: str, key: str, max_tokens: int = 32000, providers: list[str] | None = None,
        attempts: int = 4) -> dict:
    """One item through OpenRouter: the reply, the grid read from it, whether it is right, its charge, and what
    OpenRouter says of how it was served: the provider, the model it names, the generation's id and the reply's
    reasoning tokens, the only trace of the effort a provider played, since OpenRouter passes a level a model does not
    offer to the nearest it does without saying which. ``providers`` pins the call to them, in order, with no fallback."""
    import http.client
    import urllib.error
    import urllib.request

    body = {"model": model, "messages": [{"role": "user", "content": prompt(item)}],
            "reasoning": {"effort": effort}, "max_tokens": max_tokens, "usage": {"include": True}}
    if providers:
        body["provider"] = {"order": providers, "allow_fallbacks": False}
    request = urllib.request.Request(OPENROUTER, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    reply, error = None, None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                reply = json.loads(response.read())
            break
        # rate limits, dropped connections, a reply cut off mid-read, or a body that is not the JSON it should be
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException,
                json.JSONDecodeError) as exc:
            error = repr(exc)[:300]
            time.sleep(10 * 2 ** attempt)
    choice = (reply or {}).get("choices", [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    grid = parse(text)
    usage = (reply or {}).get("usage") or {}
    return {"rule": item["rule"], "k": item["k"], "correct": grid == item["output"], "parsed": grid is not None,
            "text": text[-4000:], "finish": choice.get("finish_reason"), "error": None if reply else error,
            "cost": usage.get("cost"), "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"), "max_tokens": max_tokens,
            "provider": (reply or {}).get("provider"), "served": (reply or {}).get("model"),
            "generation": (reply or {}).get("id")}


def ask_all(batch: list[dict], model: str, effort: str, key: str, max_tokens: int = 32000,
            workers: int = 16, providers: list[str] | None = None) -> list[dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda item: ask(item, model, effort, key, max_tokens, providers), batch))


def summarise(results: list[dict]) -> dict[str, dict]:
    by_rule: dict[str, dict] = {}
    for r in results:
        s = by_rule.setdefault(r["rule"], {"n": 0, "solved": 0, "unparsed": 0, "errors": 0})
        s["n"] += 1
        s["solved"] += bool(r["correct"])
        s["unparsed"] += not r["parsed"]
        s["errors"] += bool(r["error"])
    for s in by_rule.values():
        s["rate"] = s["solved"] / s["n"]
    return dict(sorted(by_rule.items()))


def run_on_modal(batch: list[dict], model: str, effort: str, hours: float, max_tokens: int,
                 workers: int, providers: list[str] | None) -> list[dict]:
    """Every item answered from one Modal container, which Modal ends at ``hours`` whatever happens here."""
    import modal

    app = modal.App("continual-arc-first-sight")
    image = (modal.Image.debian_slim(python_version=f"{sys.version_info.major}.{sys.version_info.minor}")
             .add_local_python_source("arc2_first_sight"))
    secret = modal.Secret.from_dict({"OPENROUTER_API_KEY": os.environ["OPENROUTER_API_KEY"]})

    @app.function(image=image, secrets=[secret], timeout=int(hours * 3600), cpu=2, memory=2048, serialized=True)
    def answer(payload: str) -> str:
        from arc2_first_sight import ask_all as remote_ask_all

        return json.dumps(remote_ask_all(json.loads(payload), model, effort, os.environ["OPENROUTER_API_KEY"],
                                         max_tokens, workers, providers))

    with modal.enable_output(), app.run(environment_name=os.environ.get("MODAL_ENVIRONMENT")):
        return json.loads(answer.remote(json.dumps(batch)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="openai/gpt-6-luna")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--skip", type=int, default=0, help="ask only instances from this one on, keeping the earlier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hours", type=float, default=2.0, help="Modal's timeout for the whole batch")
    parser.add_argument("--max-tokens", type=int, default=32000, help="the reply's cap, reasoning included")
    parser.add_argument("--workers", type=int, default=16, help="calls in flight at once")
    parser.add_argument("--items", type=Path, required=True, help="a track's calibration sample (JSON)")
    parser.add_argument("--providers", type=lambda v: v.split(","),
                        help="OpenRouter provider slugs every call goes to, in order, with no fallback")
    a = parser.parse_args(argv)
    sample = json.loads(a.items.read_text())
    batch = [dict(item, rule=item["task"]) for item in sample["items"] if item["k"] < a.instances]
    ids = sorted({item["rule"] for item in batch})
    prefix = f"{a.items.stem}-"
    batch = [item for item in batch if item["k"] >= a.skip]
    results = run_on_modal(batch, a.model, a.effort, a.hours, a.max_tokens, a.workers, a.providers)
    summary = summarise(results)
    folder = REPORTS
    folder.mkdir(parents=True, exist_ok=True)
    again = f"-seed{a.seed}" if a.seed else ""  # a reading recorded under another seed, beside seed 0's
    pinned = "-on-" + "-".join(slug.replace("/", "-") for slug in a.providers) if a.providers else ""
    out = folder / f"{prefix}{a.model.split('/')[-1]}-{a.effort}{again}{pinned}.json"
    previous = json.loads(out.read_text()) if out.exists() else {"results": []}
    kept = [r for r in previous["results"] if r["rule"] not in set(ids) or r["k"] < a.skip]
    merged = kept + results
    out.write_text(json.dumps({"model": a.model, "effort": a.effort, "providers": a.providers, "instances": a.instances,
                               "seed": a.seed, "n_demos": N_DEMOS, "summary": summarise(merged), "results": merged},
                              indent=1) + "\n")
    cost = sum(r["cost"] or 0 for r in results)
    solved = sum(r["correct"] for r in results)
    print(f"{a.model} {a.effort}: {solved} of {len(results)} solved over {len(summary)} rules, {cost:.2f} USD")
    for rule_id, s in summary.items():
        print(f"  {rule_id}: {s['solved']}/{s['n']}" + (f" ({s['unparsed']} unparsed)" if s["unparsed"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
