"""Fetching and loading the pinned upstream data.

Three upstream repositories are pinned by commit:

* Re-ARC (MIT) provides procedural generators and verifiers for the 400
  ARC-AGI-1 training tasks.
* ARC-AGI (Apache-2.0) provides the original 400 training and 400 evaluation
  tasks as JSON.
* ARC-AGI-2 (Apache-2.0) provides the 1,000 training and 120 evaluation tasks
  of the second edition, JSON in the same schema. No track serves them: they are
  fetched only on request (``arc2``), so that new rules, such as the held-out
  season's, can be screened against every public ARC task.

All are cloned into ``$CONTINUAL_ARC_DATA`` (default ``~/.cache/continual-arc``)
by :func:`fetch`. Nothing is vendored into this package, so the benchmark data is
exactly what the upstream authors published.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from continual_arc.grids import Grid

REARC_REPO = "https://github.com/michaelhodel/re-arc"
REARC_COMMIT = "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce"

ARC_REPO = "https://github.com/fchollet/ARC-AGI"
ARC_COMMIT = "399030444e0ab0cc8b4e199870fb20b863846f34"

ARC2_REPO = "https://github.com/arcprize/ARC-AGI-2"
ARC2_COMMIT = "f3283f727488ad98fe575ea6a5ac981e4a188e49"

ARC_SPLITS = ("training", "evaluation")

DATASET_ARC1 = "arc1"
DATASET_ARC2 = "arc2"
DATASETS = (DATASET_ARC1, DATASET_ARC2)

# Upstream name -> (repo, commit, folder under the data root). ``arc1``, the track, needs the
# first two; ``arc2`` the third.
_UPSTREAMS = {
    "re-arc": (REARC_REPO, REARC_COMMIT, "re-arc"),
    "ARC-AGI": (ARC_REPO, ARC_COMMIT, "ARC-AGI"),
    "ARC-AGI-2": (ARC2_REPO, ARC2_COMMIT, "ARC-AGI-2"),
}
_NEEDS = {DATASET_ARC1: ("re-arc", "ARC-AGI"), DATASET_ARC2: ("ARC-AGI-2",)}


class DataNotFound(RuntimeError):
    """Raised when the pinned data has not been fetched."""


def data_root() -> Path:
    override = os.environ.get("CONTINUAL_ARC_DATA")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "continual-arc"


def rearc_dir() -> Path:
    return data_root() / "re-arc"


def arc_dir() -> Path:
    return data_root() / "ARC-AGI"


def arc2_dir() -> Path:
    return data_root() / "ARC-AGI-2"


def dataset_dir(dataset: str) -> Path:
    if dataset == DATASET_ARC1:
        return arc_dir()
    if dataset == DATASET_ARC2:
        return arc2_dir()
    raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _clone_pinned(repo: str, commit: str, dest: Path, force: bool) -> None:
    if dest.exists():
        if not force and _head(dest) == commit:
            return
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", str(dest))
    _git("remote", "add", "origin", repo, cwd=dest)
    _git("fetch", "-q", "--depth", "1", "origin", commit, cwd=dest)
    _git("checkout", "-q", "FETCH_HEAD", cwd=dest)


def _head(path: Path) -> str | None:
    try:
        return _git("rev-parse", "HEAD", cwd=path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def fetch(force: bool = False, datasets: tuple[str, ...] = (DATASET_ARC1,)) -> dict[str, str]:
    """Clone the upstream repositories at their pinned commits.

    Returns a mapping of name -> local path. Safe to call repeatedly.
    """
    out: dict[str, str] = {}
    for dataset in datasets:
        for name in _NEEDS[dataset]:
            repo, commit, folder = _UPSTREAMS[name]
            dest = data_root() / folder
            _clone_pinned(repo, commit, dest, force)
            out[name] = str(dest)
    return out


def status(datasets: tuple[str, ...] = (DATASET_ARC1,)) -> dict[str, dict[str, object]]:
    """Report whether each upstream is present and at the pinned commit."""
    out: dict[str, dict[str, object]] = {}
    for dataset in datasets:
        for name in _NEEDS[dataset]:
            _, commit, folder = _UPSTREAMS[name]
            path = data_root() / folder
            head = _head(path)
            out[name] = {
                "path": str(path),
                "present": head is not None,
                "pinned": commit,
                "head": head,
                "ok": head == commit,
            }
    return out


def require_data(dataset: str = DATASET_ARC1) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")
    st = status((dataset,))
    missing = [name for name, s in st.items() if not s["ok"]]
    if missing:
        raise DataNotFound(
            f"pinned data missing or wrong commit for {missing}; "
            f"run `continual-arc data fetch --datasets {dataset}`"
        )


# ---------------------------------------------------------------------------
# ARC-AGI-1 JSON tasks
# ---------------------------------------------------------------------------

# Six known label errors in the original ARC training set, as corrected by Re-ARC's
# ``fix_bugs``. Each entry: task id, example index in ``train``, colour, cells to fill.
_TRAINING_FIXES: tuple[tuple[str, int, int, tuple[tuple[int, int], ...]], ...] = (
    ("a8d7556c", 2, 2, ((8, 12), (9, 12))),
    ("6cf79266", 2, 1, ((6, 17), (7, 17), (8, 15), (8, 16), (8, 17))),
    ("469497ad", 1, 7, ((5, 12), (5, 13), (5, 14))),
    ("9edfc990", 1, 1, ((6, 13),)),
    ("e5062a87", 1, 2, ((1, 3), (1, 4), (1, 5), (1, 6))),
    ("e5062a87", 0, 2, ((5, 2), (6, 3), (3, 6), (4, 7))),
)


def _apply_training_fixes(tasks: dict[str, dict]) -> None:
    for task_id, idx, colour, cells in _TRAINING_FIXES:
        task = tasks.get(task_id)
        if task is None:
            continue
        grid: Grid = task["train"][idx]["output"]
        for r, c in cells:
            grid[r][c] = colour


def load_arc_tasks(split: str, fix_bugs: bool = True, dataset: str = DATASET_ARC1) -> dict[str, dict]:
    """Load one split as ``{task_id: {"train": [...], "test": [...]}}``.

    Grids are ``list[list[int]]``. For the ARC-AGI-1 training split, the six
    documented label errors are corrected unless ``fix_bugs`` is False, so that the
    fixed tasks agree with the Re-ARC verifiers. ``dataset`` selects ARC-AGI-1
    (``arc1``) or ARC-AGI-2 (``arc2``); both use the same JSON schema.
    """
    if split not in ARC_SPLITS:
        raise ValueError(f"split must be one of {ARC_SPLITS}, got {split!r}")
    require_data(dataset)
    folder = dataset_dir(dataset) / "data" / split
    tasks: dict[str, dict] = {}
    for path in sorted(folder.glob("*.json")):
        with path.open() as fp:
            tasks[path.stem] = json.load(fp)
    if dataset == DATASET_ARC1 and split == "training" and fix_bugs:
        _apply_training_fixes(tasks)
    return tasks


def arc_task_ids(split: str, dataset: str = DATASET_ARC1) -> list[str]:
    require_data(dataset)
    folder = dataset_dir(dataset) / "data" / split
    return sorted(p.stem for p in folder.glob("*.json"))



