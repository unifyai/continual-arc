"""Write a sample of the second track's stream instances, from which a model's first-sight solve rate is measured.

    uv run continual-arc data fetch --datasets arc2
    uv run python scripts/arc2_calibration_sample.py [--instances 4] [--demos 3] [--seed 0] [--out FILE]

An item is one fresh instance of one of the stream tasks of ``arc2-small`` at ``--seed`` with ``--demos``
demonstrations, drawn as the held-out season's calibration drew its items: the instance at seed
``derive_seed("calibrate", seed, k)``, which no stream slot draws, and its demonstrations as the source serves the
instance's first request of that many pairs, so that a model meets the task as the demo-cache floor first does. Each
item carries the answer, the test output, beside the test input, for exact-match scoring of one answer with no
feedback. Nothing here calls a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from continual_arc.configs import config_tasks, get_config, stream_task_pool
from continual_arc.data import ARC2_COMMIT, ARCGEN_COMMIT
from continual_arc.seeds import derive_seed
from continual_arc.tasks import ArcGenSource

CONFIG = "arc2-small"


def items(instances: int, demos: int, seed: int) -> list[dict]:
    """``instances`` items of each stream task of :data:`CONFIG` at ``seed``, in task order."""
    cfg = get_config(CONFIG)
    source = ArcGenSource()
    out = []
    for task_id in config_tasks(cfg, stream_task_pool(cfg), seed):
        for k in range(instances):
            instance_seed = derive_seed("calibrate", seed, k)
            test = source.test_instance(task_id, instance_seed)
            out.append({"task": task_id, "k": k, "instance_seed": instance_seed,
                        "demos": [p.to_json() for p in source.demos(task_id, instance_seed, 0, demos, test)],
                        "input": test.input, "output": test.output})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instances", type=int, default=4, help="instances of each stream task")
    parser.add_argument("--demos", type=int, default=3, help="demonstrations of each instance")
    parser.add_argument("--seed", type=int, default=0, help="the configuration's seed, which picks its stream tasks")
    parser.add_argument("--out", type=Path, help=f"default {CONFIG}-calibration-sample-seed<seed>.json")
    a = parser.parse_args(argv)
    a.out = a.out or Path(f"{CONFIG}-calibration-sample-seed{a.seed}.json")
    sample = items(a.instances, a.demos, a.seed)
    a.out.write_text(json.dumps({
        "config": CONFIG, "seed": a.seed, "instances": a.instances, "demos": a.demos,
        "arc_gen": ARCGEN_COMMIT, "arc_agi_2": ARC2_COMMIT, "tasks": sorted({x["task"] for x in sample}),
        "items": sample,
    }) + "\n")
    print(f"wrote {len(sample)} items of {len({x['task'] for x in sample})} tasks to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
