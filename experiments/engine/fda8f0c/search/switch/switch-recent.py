"""Give a DAgger final run's own code (its node's ~/loop) the recent dose and a "started:" line that records the patience,
so that the run can go on from its next phase training on the last ten rounds' labels and as many earlier games',
with a more patient held-out stop (continual-arc-training method-search.md, "Why DAgger's phases stop"; the author, 21 Oct
2026). Each file is kept beside itself as <name>.pre-recent first. Usage: python3 switch-recent.py LOOP"""
import shutil
import sys
from pathlib import Path

loop = Path(sys.argv[1]).expanduser()


def patch(path: Path, pairs: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in pairs:
        assert text.count(old) == 1, (path.name, old[:70], text.count(old))
        text = text.replace(old, new)
    backup = path.with_name(path.name + ".pre-recent")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text)
    print(f"patched {path}")


patch(loop / "realtime.py", [
    ('''def dose(a: argparse.Namespace, k: int, init: Path | None, per_round: dict[int, list[Any]],''',
     '''# the recent dose's window: a phase trains on the last RECENT_ROUNDS rounds' labels and as many earlier games' (the
# frontier check of 21 Oct 2026, search/frontier_check.py, lowered the loss on the rounds still to come with this mix
# and not with every round's, nor with a uniform draw of the same size)
RECENT_ROUNDS = 10


def dose(a: argparse.Namespace, k: int, init: Path | None, per_round: dict[int, list[Any]],'''),
    ('''    if a.dose == "aggregate":
        return every, a.max_epochs if held_out else 1, held_out   # no held-out game yet: one pass, nothing to stop on
''',
     '''    if a.dose == "aggregate":
        return every, a.max_epochs if held_out else 1, held_out   # no held-out game yet: one pass, nothing to stop on
    if a.dose == "recent":   # the last RECENT_ROUNDS rounds' labels and as many earlier games', under the held-out stop
        since = k + 1 - RECENT_ROUNDS
        recent = [c for r in sorted(per_round) if r >= since for c in per_round[r]]
        earlier = [c for r in sorted(per_round) if r < since for c in per_round[r]]
        return (dagger.replay_plan(recent, earlier, True, a.replay_ratio, random.Random(1000 + k)),
                a.max_epochs if held_out else 1, held_out)
'''),
    ('''choices=("aggregate", "replay"), default="aggregate",''',
     '''choices=("aggregate", "replay", "recent"), default="aggregate",'''),
])
sh = loop / "realtime.sh"
text = sh.read_text()
if 'PG reward $PG_REWARD"' in text:
    patch(sh, [('PG reward $PG_REWARD"', 'PG reward $PG_REWARD, patience $PATIENCE"')])
else:
    patch(sh, [('first $FIRST_EPOCHS passes)"', 'first $FIRST_EPOCHS passes), patience $PATIENCE"')])
