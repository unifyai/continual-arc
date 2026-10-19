"""Opaque, run-specific names for tasks.

The real task id (``673ef223``) is a lookup key: the public ARC files, the Re-ARC
generators and verifiers that ``continual-arc data fetch`` downloads, and this
benchmark's own records are all indexed by it, so a learner shown the id can fetch
the answer instead of inferring the rule, and a model may recall a famous id's rule
from pretraining. Every learner the runner drives therefore sees each task under a
name derived from its id with a salt drawn once per run: stable within the run, so
memory keyed on it works, and meaningless outside it. The salt is kept in the run
directory (``alias-salt``), so a resumed run keeps its names; the records keep the
real ids, and a reader with the salt can name a task as the learner saw it.
"""

from __future__ import annotations

import hashlib
import secrets
from pathlib import Path

ALIAS_STYLES = ("hex", "names")
"""``hex``: ``task-3f2a9c1e``, the name an issue tracker would give a ticket. ``names``: a two-word
codename such as ``amber-heron``, the name a project or a client account carries in a working
setting. Both are meaningless with respect to the task's rule and salted per run."""

SALT_FILE = "alias-salt"

_CODENAME_FIRST = (
    "amber", "ashen", "birch", "bramble", "briar", "cedar", "cinder", "cobalt", "copper", "coral",
    "crimson", "dapple", "dusky", "ebony", "ember", "fallow", "fennel", "flint", "garnet", "ginger",
    "hazel", "heather", "hollow", "indigo", "ivory", "jade", "juniper", "kestrel", "lichen", "linen",
    "maple", "marble", "mossy", "nettle", "ochre", "olive", "onyx", "opal", "pebble", "pewter",
    "quartz", "rowan", "russet", "saffron", "sable", "sepia", "silver", "slate", "sorrel", "tawny",
    "teal", "thistle", "timber", "topaz", "umber", "velvet", "walnut", "willow", "wren", "yarrow",
)
_CODENAME_SECOND = (
    "anchor", "arbor", "atlas", "bay", "beacon", "bell", "bridge", "brook", "canyon", "cape",
    "compass", "cove", "crane", "crest", "dune", "falcon", "ferry", "field", "fjord", "forge",
    "gable", "glen", "grove", "harbor", "heron", "hill", "isle", "keel", "kiln", "lantern",
    "ledger", "loom", "marsh", "meadow", "mill", "moor", "orchard", "otter", "pier", "plover",
    "quarry", "quill", "reef", "ridge", "river", "saddle", "shoal", "spire", "summit", "tarn",
    "terrace", "tide", "trail", "vale", "vault", "wharf", "wick", "windrow", "yard", "zephyr",
)


def codename(digest: bytes) -> str:
    """A two-word codename from the first bytes of a digest: 60 x 60 = 3,600 names."""
    a = int.from_bytes(digest[:4], "big") % len(_CODENAME_FIRST)
    b = int.from_bytes(digest[4:8], "big") % len(_CODENAME_SECOND)
    return f"{_CODENAME_FIRST[a]}-{_CODENAME_SECOND[b]}"


class TaskAliases:
    """The run's name for each task: ``task-<8 hex>`` (or a codename) from a blake2b digest of the salt and the id."""

    def __init__(self, salt: str | None = None, style: str = "hex"):
        if style not in ALIAS_STYLES:
            raise ValueError(f"alias style must be one of {ALIAS_STYLES}, not {style!r}")
        self.salt = salt or secrets.token_hex(16)
        self.style = style
        self._cache: dict[str, str] = {}

    @classmethod
    def for_run(cls, out_dir: str | Path | None, style: str = "hex") -> TaskAliases:
        """The aliases of the run written to ``out_dir``, from its ``alias-salt``, drawn and written on the first call;
        a fresh salt when the run writes no directory."""
        if out_dir is None:
            return cls(style=style)
        path = Path(out_dir) / SALT_FILE
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(cls(style=style).salt)
        return cls(salt=path.read_text().strip(), style=style)

    def alias(self, task_id: str) -> str:
        if task_id not in self._cache:
            digest = hashlib.blake2b(f"{self.salt}:{task_id}".encode(), digest_size=8).digest()
            if self.style == "names":
                name = codename(digest)
                taken = set(self._cache.values())
                n = 2
                while name in taken:   # a collision inside one run gets a numeric suffix
                    name = f"{codename(digest)}-{n}"
                    n += 1
                self._cache[task_id] = name
            else:
                self._cache[task_id] = f"task-{digest[:4].hex()}"
        return self._cache[task_id]
