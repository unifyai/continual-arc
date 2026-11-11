"""The measured stream tokens of Figure 3's learners, which Figure 13 prices (make_figures.fig_inference_cost).

``collect_figure3_inference(mf)`` takes the make_figures module and returns (points, runs). It reads no price, makes
no model call and writes no file; the caller prices every method at one serving scenario. The tokens are measured;
the repriced dollars and the cache's share of the input are assumptions.

Only online play enters: each student's stream, and the GLM stream whose state QT, QTE and QETE play from. DAgger's
teacher labels, its optimization, validation and serving overhead are outside these token totals. The recorded output
tokens already hold the reasoning tokens, so the separate reasoning counter is never added to them when pricing.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any


# The GLM streams the teacher played for the rows whose notes it writes; the student's records do not hold those calls.
TEACHER_STREAM_ROWS = {"hg": "h", "hgpl": "hpl", "hplg": "hpl"}
TOKEN_RESOURCES = {"input": "llm_input_tokens", "output": "llm_output_tokens", "reasoning": "llm_reasoning_tokens"}


def _component(row: Any, run: Any, role: str) -> dict:
    records = [r for r in run.records if r.phase == "stream"]
    if not records:
        raise ValueError(f"{row.run}: no measured stream instances")
    tokens = {name: sum((r.resources or {}).get(field, 0) or 0 for r in records)
              for name, field in TOKEN_RESOURCES.items()}
    # Usage.as_resources does not retain the proxy's per-call cache counters. Their absence is unknown, not zero.
    tokens.update(cache_read=None, cache_write=None)
    missing = {name: sum(field not in (r.resources or {}) and (r.resources or {}).get("llm_requests", 0) > 0
                         for r in records) for name, field in TOKEN_RESOURCES.items()}
    if missing["input"] or missing["output"]:
        raise ValueError(f"{row.run}: input/output token counters absent for requests: {missing}")
    source = run.path / "records.jsonl"
    return {"role": role, "model": row.player, "run": row.run, "instances": len(records),
            "scheduled_instances": run.n_slots, "dropped_instances": len(run.dropped), "tokens": tokens,
            "requests": sum((r.resources or {}).get("llm_requests", 0) or 0 for r in records),
            "unpriced_requests": sum((r.resources or {}).get("llm_unpriced_requests", 0) or 0 for r in records),
            "token_counters_missing": missing,
            "records_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def collect_figure3_inference(mf: Any) -> tuple[list[dict], list[Any]]:
    """The raw tokens and share learned, with its interval, of each learner Figure 3 draws, and the runs read.

    A point's dollars sum all its components' repriced tokens and divide by the student's ``instances``; in the
    teacher's rows both components hold the same 200 instances. The instances an outage played through are left out
    as make_figures.played leaves them; no absent call, price or cached token is filled in.
    """
    rows = [r for r in mf.TEST_ROWS if r.row not in mf.APPENDIX_ROWS and r.row not in mf.CHANCE_ROWS
            and (r.users == 1 or r.row in mf.USER_ROWS)
            and all(mf.frozen_measure(r, test) is not None for test in mf.FROZEN_TESTS)]
    teacher_rows = {r.row: r for r in mf.TEST_ROWS if r.player == "glm" and r.users == 1}
    points, sources = [], {}
    for row in rows:
        run, learned = mf.played(row), mf.stream_learned(row)
        if run is None or not run.complete or learned is None:
            continue
        components = [_component(row, run, "student")]
        sources[str(run.path)] = run
        if row.player == "qwen" and row.row in TEACHER_STREAM_ROWS:
            teacher = teacher_rows[TEACHER_STREAM_ROWS[row.row]]
            teacher_run = mf.played(teacher)
            if teacher_run is None or not teacher_run.complete:
                raise ValueError(f"{row.run}: its teacher's stream {teacher.run} is incomplete")
            components.append(_component(teacher, teacher_run, "teacher"))
            sources[str(teacher_run.path)] = teacher_run
        totals = defaultdict(lambda: dict(input=0, output=0, reasoning=0, cache_read=None, cache_write=None))
        for component in components:
            for token in TOKEN_RESOURCES:
                totals[component["model"]][token] += component["tokens"][token]
        suffix = f"-u{row.users}" if row.users > 1 else ""
        points.append({"id": f"{row.player}-{row.row}{suffix}",
                       "name": mf.row_name(row.row, row.player) + (f"({row.users})" if row.users > 1 else ""),
                       "player": row.player, "row": row.row, "users": row.users,
                       "family": mf._row_family(row.row, row.player),
                       "learned": [learned["share_learned"], *(learned.get("ci95") or [None, None])],
                       "instances": components[0]["instances"], "scheduled_instances": run.n_slots,
                       "dropped_instances": len(run.dropped), "tokens": dict(totals), "components": components})
    return points, list(sources.values())
