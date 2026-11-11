"""generate.py RUN: a capacity run's generator, on the CPU machine that plays its games (README.md), detached and
resumable. GLM-5.3-flash plays batch after batch of the run's games (capacity.write_schedules, BATCH games a batch),
each game a fresh user in a play of its own (glm_val.py medium GAME --harness tests, under the run's alias salt), up to
the concurrency the governor allows; each batch whose games have all ended is labelled (capacity.label_batch) and its
training set left for the trainer as ROOT/sets/bKKKK.tgz, with its counts in bKKKK.json and bKKKK.ready once whole.

ROOT is ~/runs/rt/cap-RUN, which the operator's hourly backup copies off the machine:
  alias-salt        the run's task names, drawn once; the run's tests play under it too
  play/             the run's play config for the tests (frozen.py --no-memory reads alias-salt and driver.json's row)
  batches/bKKKK/    games/gNNN (each game's schedule and play), view/ and labels/ (capacity.label_batch)
  sets/             each labelled batch's training set, for the trainer
  batches.jsonl     each labelled batch's counts, a line each: its games solved and not, labels and GLM's dollars
  progress          what it did, a line each
  status.json       the governor's concurrency, games played and failed, and GLM's dollars so far
Read by it, written by the operator's capacity task (brain):
  trained           the phases the trainer has finished: no game of batch trained + AHEAD or later is started
  concurrency       the most games at once (CAP unless set): lowered while the paper's reference rows see throttling
  stop              no new game is started; it labels what has ended and exits

The governor starts at START games at once, halves on any call asked again (a 429 or a 5xx, glm_val.py's
retries.jsonl) in the last BACKOFF seconds, and adds STEP after RAISE seconds without one, up to the cap. A game that
exits without its record is played again from nothing (its folder cleared but for its schedule), at most TRIES times;
a game still playing when the generator starts, left by one that stopped, is waited for rather than played again. A
game's harness state, about 350 MB, is deleted once it has ended; its captured calls and records stay.

Run from ~/prime-rl with the bundle's ~/loop on the path and the key in ~/.openrouter.env (launch.sh starts it):
  cd ~/prime-rl && PYTHONPATH=$HOME/loop setsid nohup uv run --no-sync python ~/gen/tools/capacity/generate.py t25 &
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import capacity

HOME = Path.home()
TOOLS = Path(__file__).resolve().parents[1]   # ~/gen/tools, where launch.sh puts glm_val.py beside capacity/
START, STEP, CAP = 4, 2, 16
BACKOFF, RAISE = 600, 1800
AHEAD = 3
TRIES = 3
POLL = 15


def root_of(run: str) -> Path:
    return HOME / "runs/rt" / f"cap-{run}"


class Generator:
    def __init__(self, run: str):
        self.run, self.root = run, root_of(run)
        self.batches, self.sets = self.root / "batches", self.root / "sets"
        self.sets.mkdir(parents=True, exist_ok=True)
        self.running: dict[Path, subprocess.Popen | int] = {}   # a game's play, or the pid of one adopted
        self.started: dict[Path, float] = {}
        self.k, self.changed = START, time.time()
        self.labelling: threading.Thread | None = None
        self.ids = capacity.tasks(run)
        self.made: dict[int, list[Path]] = {}

    def log(self, message: str) -> None:
        with open(self.root / "progress", "a") as f:
            f.write(f"{datetime.now(timezone.utc):%m-%d %H:%M} generate.py {self.run}: {message}\n")

    def number(self, name: str, default: int) -> int:
        f = self.root / name
        return int(f.read_text().strip()) if f.exists() and f.read_text().strip() else default

    # -- the run's own files --

    def prepare(self) -> None:
        """The run's alias salt, drawn once, and the play config its tests read (README.md, "The tests")."""
        salt = self.root / "alias-salt"
        if not salt.exists():
            salt.write_text(secrets.token_hex(16))
        play = self.root / "play"
        play.mkdir(exist_ok=True)
        shutil.copy2(salt, play / "alias-salt")
        if not (play / "driver.json").exists():
            sched = capacity.schedule(self.run, 0, 0, *capacity.instances(self.run, self.ids, 0)[0])
            (play / "driver.json").write_text(json.dumps(
                {"row": "dagger", "capacity": self.run, "tasks": capacity.RUNS[self.run],
                 "note": "the student's tests play the DAgger row's harness, its weights alone, under this salt"},
                indent=1))
            for f in ("config.json", "meta.json"):
                (play / f).write_text(json.dumps(sched.meta, indent=1))
            sched.save(play / "schedule.json")

    # -- games --

    def games(self, k: int) -> list[Path]:
        if k not in self.made:
            self.made[k] = capacity.write_schedules(self.run, k, self.batches / f"b{k:04d}" / "games", self.ids)
        return self.made[k]

    @staticmethod
    def ended(game: Path) -> bool:
        return (game / "records.jsonl").exists() and (game / "records.jsonl").read_text().strip() != ""

    @staticmethod
    def failed(game: Path) -> bool:
        return (game / "failed").exists()

    def tries(self, game: Path) -> int:
        f = game / "tries"
        return int(f.read_text()) if f.exists() else 0

    def start(self, game: Path) -> None:
        """A game from nothing: its folder cleared but for its schedule and tries, then glm_val.py, detached."""
        for p in game.iterdir():
            if p.name not in ("schedule.json", "tries"):
                shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
        (game / "tries").write_text(str(self.tries(game) + 1))
        cmd = ["uv", "run", "--no-sync", "--env-file", str(HOME / ".openrouter.env"), "python",
               str(TOOLS / "glm_val.py"), "medium", str(game), "--schedule", str(game / "schedule.json"),
               "--alias-salt", str(self.root / "alias-salt"), "--harness", "tests"]
        with open(game / "play.log", "a") as log:
            self.running[game] = subprocess.Popen(cmd, cwd=HOME / "prime-rl", stdout=log, stderr=subprocess.STDOUT,
                                                  stdin=subprocess.DEVNULL, start_new_session=True)
        self.started[game] = time.time()

    def adopt(self) -> None:
        """The run's games still playing, left by a generator that stopped: waited for as its own."""
        out = subprocess.run(["pgrep", "-af", "glm_val.py medium"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            pid, *argv = line.split()
            game = Path(argv[argv.index("medium") + 1]) if "medium" in argv else None
            if game and game.is_relative_to(self.batches) and game not in self.running and argv[0].endswith("python"):
                self.running[game] = int(pid)
                self.started[game] = time.time()
        if self.running:
            self.log(f"{len(self.running)} games still playing from before, waited for")

    @staticmethod
    def alive(proc: subprocess.Popen | int) -> bool:
        if isinstance(proc, int):
            try:
                os.kill(proc, 0)
                return True
            except ProcessLookupError:
                return False
        return proc.poll() is None

    def reap(self) -> None:
        for game, proc in list(self.running.items()):
            if self.alive(proc):
                continue
            del self.running[game]
            if self.ended(game):
                shutil.rmtree(game / "users" / "u000" / "system" / "state", ignore_errors=True)
            elif self.tries(game) >= TRIES:
                (game / "failed").touch()
                self.log(f"{game.relative_to(self.root)} failed {TRIES} times, left out (its play.log)")
            else:
                code = proc if isinstance(proc, int) else proc.returncode
                self.log(f"{game.relative_to(self.root)} ended without its record ({code}): played again")

    def waiting(self) -> list[Path]:
        """The games to start next, in order: every unstarted game of the batches the trainer allows."""
        out, last = [], self.number("trained", 0) + AHEAD
        k = 0
        while k < last:
            games = self.games(k)
            out += [g for g in games if g not in self.running and not self.ended(g) and not self.failed(g)]
            k += 1
            if len(out) >= CAP * 2:
                break
        return out

    # -- the governor --

    def retried(self, since: float) -> int:
        n = 0
        for game, t in self.started.items():
            if t < since - 86_400:
                continue
            f = game / "retries.jsonl"
            if f.exists():
                n += sum(json.loads(line)["t"] >= since for line in f.read_text().splitlines() if line.strip())
        return n

    def govern(self) -> None:
        cap, now = self.number("concurrency", CAP), time.time()
        if self.retried(now - BACKOFF) and now - self.changed >= BACKOFF:
            self.k, self.changed = max(1, self.k // 2), now
            self.log(f"calls asked again in the last {BACKOFF // 60} minutes: {self.k} games at once")
        elif now - self.changed >= RAISE and self.k < cap and not self.retried(now - RAISE):
            self.k, self.changed = min(cap, self.k + STEP), now
            self.log(f"no call asked again for {RAISE // 60} minutes: {self.k} games at once")
        self.k = min(self.k, cap)

    # -- labels --

    def label_ready(self) -> None:
        """The oldest played batch without a training set, labelled in a thread of its own, one at a time."""
        if self.labelling and self.labelling.is_alive():
            return
        k = 0
        while (self.batches / f"b{k:04d}").exists():
            name = f"b{k:04d}"
            games = self.games(k)
            if not (self.sets / f"{name}.ready").exists():
                if all(self.ended(g) or self.failed(g) for g in games):
                    self.labelling = threading.Thread(target=self.label, args=(k,), daemon=True)
                    self.labelling.start()
                return
            k += 1

    def label(self, k: int) -> None:
        name = f"b{k:04d}"
        batch = self.batches / name
        try:
            shutil.rmtree(batch / "labels" / "set", ignore_errors=True)
            counts = capacity.label_batch(batch)
            tmp = self.sets / f"{name}.tgz.tmp"
            subprocess.run(["tar", "-C", str(batch / "labels"), "--transform", f"s,^set,{name},", "-czf", str(tmp),
                            "set"], check=True)
            tmp.rename(self.sets / f"{name}.tgz")
            (self.sets / f"{name}.json").write_text(json.dumps({"batch": k, **counts}, indent=1))
            with open(self.root / "batches.jsonl", "a") as f:
                f.write(json.dumps({"at": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}", "batch": k, **counts}) + "\n")
            (self.sets / f"{name}.ready").touch()
            self.log(f"{name} labelled: {json.dumps(counts)}")
        except Exception as e:  # noqa: BLE001  (a batch that will not label is said and tried again next pass)
            self.log(f"{name} could not be labelled: {type(e).__name__}: {e}")
            time.sleep(300)

    def status(self) -> None:
        played = failed = 0
        for b in sorted(self.batches.glob("b*")):
            for g in (b / "games").glob("g*"):
                played += self.ended(g)
                failed += self.failed(g)
        spent = sum(json.loads(f.read_text()).get("cost_usd", 0) for f in self.sets.glob("b*.json"))
        (self.root / "status.json").write_text(json.dumps(
            {"at": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}", "concurrency": self.k, "running": len(self.running),
             "games_played": played, "games_failed": failed, "sets": len(list(self.sets.glob("b*.ready"))),
             "glm_usd_labelled": round(spent, 2), "trained": self.number("trained", 0),
             "stopped": (self.root / "stop").exists()}, indent=1))

    def loop(self) -> None:
        self.prepare()
        self.adopt()
        self.log(f"started: {capacity.RUNS[self.run]}'s {len(self.ids)} tasks, {capacity.BATCH} games "
                 f"a batch, {self.k} at once to start, at most {self.number('concurrency', CAP)}")
        while True:
            self.reap()
            stop = (self.root / "stop").exists()
            if not stop:
                self.govern()
                for game in self.waiting()[:max(0, self.k - len(self.running))]:
                    self.start(game)
            self.label_ready()
            self.status()
            if stop and not self.running and not (self.labelling and self.labelling.is_alive()):
                self.log("stopped: no game is running and every ended batch is labelled")
                return
            time.sleep(POLL)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", choices=capacity.RUNS)
    Generator(p.parse_args().run).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
