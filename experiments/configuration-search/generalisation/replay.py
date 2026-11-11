"""replay.py VIEW OUT FIRST LAST INIT: a DAgger run's training replayed offline from INIT, the adapter of a phase it
trained under the rule it trained by before its switch to the recent dose (every label so far, the held-out stop at
patience 2), through phases FIRST..LAST under that rule, on the label sets the run itself collected, so that the one
difference from the run's own phases is the rule it trained by after the switch (the author, 25 Oct 2026: "maybe we continue
another run that goes back to the switch point and continues using the original rule"). Each phase is the run's own
train_phase, from the DAgger rows' realtime.py, warm-started from the phase before; VIEW is a folder whose
play/rounds/rNNNN/ hold each round's labels/split and its captured games by name (cell.py's run_inputs). Resumable: a
phase whose OUT/pNNNN/DONE exists is taken as done. Prints each phase's summary as a JSON line.

Run (from ~/prime-rl, the DAgger rows' code on the path; cell.py's replay kind runs it):
  PYTHONPATH=$HOME/loop uv run --no-sync python replay.py ~/gen/NAME/view ~/gen/NAME/phases 69 135 ~/gen/NAME/phases/p0068
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scratch import TRAINER_ARGV   # the DAgger final runs' trainer arguments before the switch


def main() -> None:
    view, out, first, last, init = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    import realtime
    import train
    seen = []
    realtime.cmd_trainer = seen.append   # main binds each subcommand when it runs: the trainer's is caught, not run
    sys.argv = ["realtime.py", "trainer", "--run", str(view), *TRAINER_ARGV, "--gpus", "8"]
    realtime.main()
    (a,) = seen
    targs = realtime.train_args(a)
    trainer = train.make_trainer(targs)
    adapter = init / "adapter"
    for k in range(first, last + 1):
        phase = out / f"p{k:04d}"
        if not (phase / "DONE").exists():
            phase.mkdir(parents=True, exist_ok=True)
            done = realtime.train_phase(a, view, targs, trainer, k, phase, adapter)
            print(json.dumps({"phase": k, **{key: done.get(key) for key in ("steps", "minutes", "best_step")}}), flush=True)
        adapter = phase / "adapter"


if __name__ == "__main__":
    main()
