"""A submitted learner served from a machine of its own.

The verified tier's rules and the submitter's key stay on the verifier's machine, and the submission's code runs on a
disposable one, the arena, that holds neither and has no way out but this connection: its sandbox
(:class:`~continual_arc_baselines.systems.hosted.HostedLearner`) can be broken, and then what it reaches is an empty
machine. :class:`RemoteLearner` is the verifier's side. It starts this module on the arena over ssh, speaks the learner
protocol of :mod:`learner_host` over the connection, and carries the arena's calls to the tracking proxy, on the
verifier's machine, through a port the connection forwards. The model calls are counted here, by the proxy, and every
line from the arena is untrusted and bounded (:class:`~continual_arc_baselines.systems.hosted.ServedLearner`).

When the run ends, what the learner printed and what it kept come back as bytes capped in size, never unpacked here:
``learner.log`` and ``kept.tar``, which stay private with the run's records.

Between two instances of a frozen test the arena reboots. Putting a learner's files back and ending its processes leaves the
kernel as the instance left it: its counters of processes forked and of pipes and sockets made, which any later process can
read and an earlier one can set, its page cache, its free memory. So before each instance :meth:`RemoteLearner.reset` ends the
connection, reboots the machine, takes what the learner printed off it, since a file that grows is a number the next instance
could read in the disk's free space, and only then connects again, to a host that starts from a copy of the stream's files.
What outlasts a reboot is the disk, and everything of the learner's on it has been put back. An instance of a frozen test on
an arena therefore has nothing of the last but the clock: it can end later, and the next can read the time.

Run on the arena::

    python -m continual_arc_baselines.systems.remote check
    python -m continual_arc_baselines.systems.remote serve --python VENV/bin/python --learner pkg.module:Class \\
        --code DIR --workdir DIR --base-url http://127.0.0.1:PORT/v1 --timeouts 3600,600,60

An arena is an ``arena`` user on a machine no other run has used, with its baselines' environment at
:data:`ARENA_PYTHON` and its run under :data:`ARENA_ROOT`. ``check`` is what the verifier asks before anything of a
submission's runs on it and again before any instance does: that nothing outside answers it and that its sandbox holds
(:func:`arena_problems`), and the machine's firewall, outside it, is what the operator that rents it checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from continual_arc_baselines.frozen import delete
from continual_arc_baselines.systems.hosted import TMP_PREFIX, HostedLearner, HostedLearnerError, ServedLearner
from continual_arc_baselines.systems.sandbox import confinement_problems, proxy_port

ARENA_ROOT = "/home/arena/run"
ARENA_PYTHON = "/home/arena/baselines/bin/python"
ARENA_UV_PATH = "/home/arena/.local/bin:/usr/bin:/bin"
LOG_CAP = 2**30
TEST_LOG_CAP = 2**23  # of what the learner prints in one instance of a frozen test
KEPT_CAP = 2 * 2**30  # a harness's state is a few hundred megabytes (frozen.py), and a truncated tar reads up to its cut
FETCH_TIMEOUT_S = 3600
REBOOT = "/usr/sbin/reboot"
REBOOT_TIMEOUT_S = 600
MAX_INSTALLED = 2000
INSTALLED_NAME = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]{0,99})-([A-Za-z0-9][A-Za-z0-9.!+_-]{0,59})\.dist-info")
# Every option that lets the far side of an ssh connection act on this one is off, and no configuration file is read.
SSH_OPTIONS = ("-F", "/dev/null", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", "StrictHostKeyChecking=accept-new", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
               "-o", "PermitLocalCommand=no", "-o", "UpdateHostKeys=no", "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=20", "-o", "ExitOnForwardFailure=yes", "-o", "LogLevel=ERROR")


def ssh_command(arena: str, key: str | Path, known_hosts: str | Path, command: str,
                forward: tuple[int, int] | None = None, max_bytes: int = LOG_CAP, connect_s: int = 60) -> list[str]:
    """ssh to ``arena`` to run ``command``; ``forward`` opens the arena's loopback port ``forward[0]`` onto this machine's
    loopback port ``forward[1]``. A file ssh writes, its output or its errors, stops at ``max_bytes`` however much the far
    side sends, since the far side is not to be trusted (where ``prlimit`` is, as on any Linux)."""
    tunnel = ["-R", f"127.0.0.1:{forward[0]}:127.0.0.1:{forward[1]}"] if forward else []
    limit = ["prlimit", f"--fsize={max_bytes}"] if shutil.which("prlimit") else []
    return [*limit, "ssh", *SSH_OPTIONS, "-o", f"ConnectTimeout={connect_s}", "-i", str(key),
            "-o", f"UserKnownHostsFile={known_hosts}", *tunnel, arena, command]


def _clean_env() -> dict[str, str]:
    """All an ssh to the arena needs of this process's environment, which holds the submitter's key."""
    return {name: os.environ[name] for name in ("PATH", "HOME", "LANG") if name in os.environ}


class RemoteLearner(ServedLearner):
    """On the verifier's machine, the learner :func:`main` hosts on ``arena`` (``user@address``, reached with ``key``,
    its host key kept in ``known_hosts``), from the environment an install made there: ``python``, ``code`` and ``runtime``
    as :class:`HostedLearner` takes them, ``arena_workdir`` the cell's own folder there, and ``arena_python`` the
    interpreter of the arena's baselines. ``margin_s`` is how much later than the arena's own limits this side gives up."""

    def __init__(self, arena: str, key: str, known_hosts: str, python: str, learner: str, arena_workdir: str,
                 arena_python: str = ARENA_PYTHON, code: str | None = None, runtime: list[str] | None = None, timeout_s: float = 3600.0,
                 startup_timeout_s: float = 600.0, close_timeout_s: float = 60.0, margin_s: float = 300.0):
        super().__init__(timeout_s + margin_s, startup_timeout_s + margin_s, close_timeout_s + margin_s)
        self.arena, self.key, self.known_hosts = arena, key, known_hosts
        self.python, self.learner, self.code, self.runtime = python, learner, code, runtime
        self.arena_workdir, self.arena_python = arena_workdir, arena_python
        self.inner_timeouts = (timeout_s, startup_timeout_s, close_timeout_s)
        self.base_url: str | None = None
        self.frozen = False  # a frozen test has begun: every reset reboots the arena (this module's docstring)

    def attach(self, workdir: Path, environ: dict[str, str], tracker: Any = None) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker
        self.base_url = environ.get("ARC_LLM_BASE_URL") or environ.get("OPENAI_BASE_URL")
        self._log = (self.workdir / "connection.log").open("a")
        try:
            self._connect()
        except BaseException:
            self.close()
            raise

    def _connect(self) -> None:
        port = proxy_port(self.base_url)
        options = ["--python", self.python, "--learner", self.learner, "--workdir", self.arena_workdir,
                   "--base-url", self.base_url, "--timeouts", ",".join(f"{t:g}" for t in self.inner_timeouts),
                   *(["--frozen"] if self.frozen else []),
                   *(["--code", self.code] if self.code else []), *(x for p in self.runtime or () for x in ("--runtime", p))]
        command = shlex.join([self.arena_python, "-m", "continual_arc_baselines.systems.remote", "serve", *options])
        self._serve_from(ssh_command(self.arena, self.key, self.known_hosts, command, forward=(port, port)), env=_clean_env())
        self._note(f"on {self.arena}: {self.hello.get('learner')} (python {self.hello.get('python')}, continual_arc "
                   f"{self.hello.get('continual_arc')})")

    def freeze(self) -> None:
        self._call("freeze")
        self.frozen = True

    def reset(self) -> None:
        if not self.frozen:
            self._call("reset")
            return
        self._kill()
        self._reboot()
        where = shlex.quote(f"{self.arena_workdir}/learner.log")
        self._fetch(f"cat {where}", self.workdir / "learner.log", TEST_LOG_CAP, then=f"rm -f {where}")
        self._connect()
        if self._round is not None:
            # the runner begins a round before it resets for the round's instance, so the new learner joins it here
            self._call("begin_round", n=self._round)

    def _boot(self) -> str | None:
        """Which boot the arena is on, or None while it does not answer."""
        try:
            done = subprocess.run(ssh_command(self.arena, self.key, self.known_hosts, "head -c 64 /proc/sys/kernel/random/boot_id",
                                              connect_s=5), capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                  timeout=30, check=False, env=_clean_env())
        except subprocess.TimeoutExpired:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    def _reboot(self) -> None:
        before = self._boot()
        if before is None:
            raise self._fail(f"the arena {self.arena} does not answer before its reboot")
        subprocess.run(ssh_command(self.arena, self.key, self.known_hosts, f"sudo -n {REBOOT}", connect_s=5), capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=60, check=False, env=_clean_env())
        deadline = time.monotonic() + REBOOT_TIMEOUT_S
        while time.monotonic() < deadline:
            now = self._boot()
            if now is not None and now != before:
                return
            time.sleep(1)
        raise self._fail(f"the arena {self.arena} did not come back from a reboot in {REBOOT_TIMEOUT_S} s")

    def describe(self) -> dict[str, Any]:
        return {"system": "remote", "arena": self.arena, "learner": self.learner, "timeout_s": self.timeout_s,
                "host": dict(self.hello), "hosted": self._call("describe") if self._proc is not None else None}

    def _fetch(self, command: str, to: Path, cap: int, then: str = "true") -> None:
        """What ``command`` prints on the arena, at most ``cap`` bytes of it, on the end of ``to``, which stops at twice its
        own cap whatever the arena sends: bytes, whatever they hold. ``then`` runs there after it."""
        with to.open("ab") as out:
            subprocess.run(ssh_command(self.arena, self.key, self.known_hosts, f"{command} | head -c {cap}; {then}",
                                       max_bytes=2 * max(cap, LOG_CAP)),
                           stdout=out, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=FETCH_TIMEOUT_S,
                           check=False, env=_clean_env())

    def close(self) -> None:
        """Ask the arena to end the learner, whose errors are noted rather than raised, since this runs while the run
        unwinds; then bring back what the learner printed and kept."""
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._call("close", timeout_s=self.close_timeout_s)
                self._proc.wait(timeout=self.close_timeout_s)
        except (HostedLearnerError, subprocess.TimeoutExpired) as error:
            self._note(f"did not close cleanly: {error}")
        finally:
            self._kill()
        where = shlex.quote(self.arena_workdir)
        try:
            self._fetch(f"cat {where}/learner.log", self.workdir / "learner.log", LOG_CAP)
            self._fetch(f"tar -C {where} -cf - --ignore-failed-read state home stream-end", self.workdir / "kept.tar",
                        KEPT_CAP)
        except (OSError, subprocess.TimeoutExpired) as error:
            self._note(f"what the learner printed and kept did not all come back: {error}")
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None


# Places outside that an arena must not reach: big public resolvers by address, its host's own resolvers, and the code host.
OUTSIDE_TCP = (("1.1.1.1", 443), ("8.8.8.8", 53), ("9.9.9.9", 443), ("185.12.64.1", 53), ("140.82.112.3", 443),
               ("2606:4700:4700::1111", 443))
OUTSIDE_UDP = ("8.8.8.8", "1.1.1.1", "185.12.64.1", "185.12.64.2")
DNS_QUERY = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01"
PROBE_S = 3.0


def arena_problems(extra_outside: tuple[tuple[str, int], ...] = ()) -> list[str]:
    """What is wrong with this machine as an arena: anything outside it that answers, the private package on it, or a
    sandbox that holds less than it should."""
    problems = []
    for host, port in (*OUTSIDE_TCP, *extra_outside):
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(PROBE_S)
            if sock.connect_ex((host, port)) == 0:
                problems.append(f"{host}:{port} answered")
    for host in OUTSIDE_UDP:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(PROBE_S)
            try:
                sock.sendto(DNS_QUERY, (host, 53))
                sock.recv(512)
                problems.append(f"{host} answered DNS")
            except OSError:
                pass
    try:
        socket.getaddrinfo("github.com", 443)
        problems.append("github.com resolves")
    except OSError:
        pass
    if importlib.util.find_spec("continual_arc_private") is not None:
        problems.append("the private package is installed")
    if subprocess.run(["sudo", "-n", "-l", REBOOT], capture_output=True, check=False).returncode != 0:
        problems.append("it cannot reboot between the instances of a frozen test")
    if subprocess.run(["sudo", "-n", "true"], capture_output=True, check=False).returncode == 0:
        problems.append("its user may run anything as root")
    ceiling = subprocess.run(["systemctl", "show", "-p", "MemoryMax", "-p", "TasksMax", f"user-{os.getuid()}.slice"],
                             capture_output=True, text=True, check=False).stdout.split()
    if sorted(line.partition("=")[0] for line in ceiling) != ["MemoryMax", "TasksMax"] or any(
            line.endswith("=infinity") for line in ceiling):
        problems.append("its user has no ceiling on memory and tasks, so a bomb could take the machine")
    with tempfile.TemporaryDirectory() as folder:
        problems += confinement_problems(Path(folder) / "filter")
    return problems


def installed(venv: Path) -> list[str]:
    """``name==version`` of what a virtual environment holds, from the names of its ``.dist-info`` directories alone, kept
    to plain letters and digits: nothing in them is read, and this is read before any instance has been played, so it holds
    nothing held out."""
    found = []
    for site in sorted(venv.glob("lib/python*/site-packages")):
        found += [f"{m[1]}=={m[2]}" for m in map(INSTALLED_NAME.fullmatch, sorted(os.listdir(site))[:100_000]) if m]
    return found[:MAX_INSTALLED]


def _serve(args: argparse.Namespace) -> int:
    """A :class:`HostedLearner` served as :mod:`learner_host` serves a learner, on stdin and stdout."""
    from continual_arc_baselines.systems.learner_host import _failure, _send, serve

    requests = os.fdopen(os.dup(0), "r", encoding="utf-8")
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8")
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    timeout_s, startup_timeout_s, close_timeout_s = (float(t) for t in args.timeouts.split(","))
    learner = HostedLearner(args.python, args.learner, code=args.code, runtime=args.runtime, timeout_s=timeout_s,
                            startup_timeout_s=startup_timeout_s, close_timeout_s=close_timeout_s)
    if args.frozen:  # what the last instance's sandbox kept in its temporary directory is space on the disk the next can read
        for left in Path(tempfile.gettempdir()).glob(TMP_PREFIX + "*"):
            delete(left)
    try:
        learner.attach(Path(args.workdir), {"ARC_LLM_BASE_URL": args.base_url}, frozen=args.frozen)
    except Exception as error:
        _send(replies, _failure(error))
        return 1
    _send(replies, {"ok": True, "result": dict(learner.hello)})
    try:
        serve(learner, requests, replies)
    finally:
        learner.abandon()  # after a close asked for, a second that does nothing
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="what is wrong with this machine as an arena, as JSON")
    check.add_argument("--outside", action="append", default=[], metavar="HOST:PORT", help="another place it must not reach")
    served = sub.add_parser("serve", help="host a learner on stdin and stdout")
    served.add_argument("--python", required=True)
    served.add_argument("--learner", required=True)
    served.add_argument("--workdir", required=True)
    served.add_argument("--base-url", required=True)
    served.add_argument("--timeouts", required=True, help="seconds: a call's, the start's and the close's")
    served.add_argument("--code")
    served.add_argument("--runtime", action="append")
    served.add_argument("--frozen", action="store_true", help="in a frozen test, from a copy of the files its stream left")
    forget = sub.add_parser("forget", help="delete a cell's folder, whatever its learner did to its permissions")
    forget.add_argument("--workdir", required=True)
    listed = sub.add_parser("installed", help="what a virtual environment holds, as JSON")
    listed.add_argument("--venv", required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    if args.command == "forget":
        delete(Path(args.workdir))
        return 0
    if args.command == "installed":
        print(json.dumps({"installed": installed(Path(args.venv))}))
        return 0
    outside = tuple((host, int(port)) for host, _, port in (o.rpartition(":") for o in args.outside))
    print(json.dumps({"problems": arena_problems(outside)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
